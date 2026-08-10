import textwrap
from datetime import date
from typing import Any, List

from writer.abstract import register_abstract_template
from writer.blocks.base_block import WriterBlock
from writer.ss_types import AbstractTemplate

DEFAULT_MODEL = "palmyra-x5"


def _content_to_text(content: Any) -> str:
    """Flatten a message's content (str or list of fragments) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for fragment in content:
            if isinstance(fragment, str):
                parts.append(fragment)
            elif isinstance(fragment, dict):
                parts.append(str(fragment.get("text") or ""))
        return " ".join(p for p in parts if p)
    return str(content or "")


class WriterToolCalling(WriterBlock):
    @classmethod
    def register(cls, type: str):
        super(WriterToolCalling, cls).register(type)
        register_abstract_template(
            type,
            AbstractTemplate(
                baseType="blueprints_node",
                writer={
                    "name": "Tool calling",
                    "description": "Connects the Agent to external tools to complete tasks it cannot handle directly.",
                    "category": "Writer",
                    "fields": {
                        "prompt": {
                            "name": "Prompt",
                            "type": "Text",
                            "control": "Textarea",
                            "desc": "The task that needs to be carried out.",
                        },
                        "modelId": {"name": "Model", "type": "Model Id", "default": DEFAULT_MODEL},
                        "maxIterations": {
                            "name": "Max iterations",
                            "type": "Number",
                            "default": 10,
                        },
                        "tools": {
                            "name": "Tools",
                            "type": "Tools",
                            "default": "{}",
                            "init": "",
                            "category": "Tools",
                        },
                        "toolGating": {
                            "name": "Tool gating",
                            "type": "Text",
                            "control": "Toggle",
                            "default": "no",
                            "desc": "Score each tool's schema against the current turn and inject only the most relevant ones, reducing the per-turn Tools Tax. Off preserves eager behavior.",
                            "category": "Tools",
                        },
                        "maxToolsPerTurn": {
                            "name": "Max tools per turn",
                            "type": "Number",
                            "default": 5,
                            "desc": "When tool gating is on, how many tools (besides the control tool) to expose to the model each turn.",
                            "category": "Tools",
                        },
                    },
                    "outs": {
                        "tools": {
                            "name": "Tools",
                            "field": "tools",
                            "description": "Run associated tools.",
                            "style": "dynamic",
                        },
                        "success": {
                            "name": "Success",
                            "description": "The task was completed successfully.",
                            "style": "success",
                        },
                        "error": {
                            "name": "Error",
                            "description": "There was an error completing the task.",
                            "style": "error",
                        },
                    },
                },
            ),
        )

    def _make_callable(self, tool_name: str):
        import time

        def callable(**args):
            expanded_execution_environment = self.execution_environment | args
            raw_return_value = self.runner.run_branch(
                self.component.id,
                f"tools_{tool_name}",
                expanded_execution_environment,
                f"Blueprint branch execution (tool {tool_name})",
            )

            if raw_return_value is None:
                self.outcome = "error"
                raise ValueError(
                    f'No value has been returned for the outcome branch "{tool_name}". Use the block "Return value" to specify one.'
                )

            transformed_result = self._project_common_tools_result(raw_return_value)
            if transformed_result is None:
                # Fallback avoids returning the literal string "None"
                transformed_result = raw_return_value
            return_value = repr(transformed_result)

            trace = self.execution_environment.get("trace")
            if trace is not None:
                trace.append(
                    {
                        "type": "functionCall",
                        "time": time.time(),
                        "name": tool_name,
                        "parameters": args,
                    }
                )
            return return_value

        return callable

    def _get_tools(self):
        import writer.ai

        tools_raw = self._get_field("tools", True)
        tools = []

        for tool_name, tool_raw in tools_raw.items():
            tool_type = tool_raw.get("type")
            tool = None
            if tool_type == "function":
                tool = writer.ai.FunctionTool(
                    type="function",
                    name=tool_name,
                    description=tool_raw.get("description"),
                    callable=self._make_callable(tool_name),
                    parameters=tool_raw.get("parameters"),
                )
            elif tool_type == "graph":
                tool = writer.ai.GraphTool(
                    type="graph",
                    graph_ids=tool_raw.get("graph_ids"),
                    subqueries=False,
                    description=tool_name,
                )
            else:
                continue
            tools.append(tool)

        def reasoning_callable(**kwargs):
            import time

            thought = kwargs.get("thought")
            action = kwargs.get("action")
            status = kwargs.get("status")

            trace = self.execution_environment.get("trace")
            if trace is not None:
                trace.append(
                    {"type": "reasoning", "time": time.time(), "thought": thought, "action": action}
                )
            if status == "DONE":
                self.is_complete = True

        reasoning_tool = {
            "type": "function",
            "name": "disclose_reasoning",
            "description": "Use this only to briefly summarize your reasoning and to signal completion. Call this when you're ready to finalize your answer.",
            "callable": reasoning_callable,
            "parameters": {
                "thought": {
                    "type": "string",
                    "description": "A brief summary of your reasoning process.",
                },
                "action": {
                    "type": "string",
                    "description": "When status is DONE, always set this to 'finalize_and_answer'. Otherwise, summarize the actions you took and why.",
                },
                "status": {
                    "type": "string",
                    "description": "Set to DONE if you consider the task complete and are ready to provide the final answer. Set to INCOMPLETE if you wish to keep iterating.",
                },
            },
        }

        tools.append(reasoning_tool)

        return tools

    def _current_intent(self) -> str:
        """Best available signal for what the agent should do this turn.

        Prefers the most recent conversational message (the live turn intent
        -- e.g. the assistant's latest plan after a tool result); falls back
        to the original task prompt when no message is available yet.
        """
        conversation = getattr(self, "_conversation", None)
        messages = getattr(conversation, "messages", None)
        if messages:
            last = messages[-1]
            content = (
                last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
            )
            if content:
                return _content_to_text(content)
        prompt = self._get_field("prompt", required=False)
        return str(prompt or "")

    def _select_tools(self, tools):
        """Gate ``tools`` to the most relevant subset for the current turn.

        Implements the Tool Attention gating step (adapted from
        arXiv:2604.21816v1): score every tool's schema against the current
        intent and return only the top-k, instead of eagerly injecting all
        schemas every turn. Control tools (e.g. ``disclose_reasoning``) are
        always kept so the ReAct loop can still finalize.
        """
        from writer.blocks.tool_gating import select_tools

        max_tools = int(self._get_field("maxToolsPerTurn", False, "5"))
        return select_tools(self._current_intent(), tools, max_tools)

    def _get_react_prompt(self, base_prompt: str):
        return textwrap.dedent(f"""
            You're a ReAct agent. Your knowledge cut-off date is 2024, but today is {str(date.today())}.
            
            Task: {base_prompt.strip()}
            
            ## Available Tools
            You have access to various tools to help complete the task. Use them as needed to gather information.
            - `disclose_reasoning` - Use this only to briefly summarize your reasoning and to signal completion
            
            ## Finalization Protocol (CRITICAL)
            When your analysis is complete, follow these steps EXACTLY:
            
            1. Make one final function call to `disclose_reasoning` with:
            ```json
            {{
                "thought": "<brief reasoning summary>",
                "action": "finalize_and_answer",
                "status": "DONE"
            }}
            ```
            
            2. **IMMEDIATELY AFTER** that tool call, output a normal assistant message (no further tool calls) containing your complete, final answer to the user.
            
            3. **DO NOT** end with a null/None action. Always provide the full answer in your final assistant message.
            
            If more information is still needed, call `disclose_reasoning` with `"status": "INCOMPLETE"` and continue gathering information.
            
            ## Important Notes
            - Use tools to gather all necessary information before finalizing
            - The `action` field in `disclose_reasoning` should always be set to "finalize_and_answer" when status is "DONE"
            - Your final assistant message (after the final `disclose_reasoning` call) should contain the complete response to the user
        """).strip()

    def _project_common_tools_result(self, tool_result):
        # Handle common HTTP-like shapes safely; pass through other types.
        if isinstance(tool_result, dict):
            if tool_result.get("request") and "body" in tool_result:
                return tool_result["body"]
        return tool_result

    def run(self):
        import writer.ai

        self.is_complete = False
        self.result = None

        try:
            prompt = self._get_field("prompt")
            model_id = self._get_field("modelId", False, default_field_value=DEFAULT_MODEL)
            max_iterations = max(1, int(self._get_field("maxIterations", False, "10")))
            conversation = writer.ai.Conversation()
            self._conversation = conversation
            tools = self._get_tools()
            tool_gating_enabled = self._get_field("toolGating", False, "no") == "yes"

            conversation += {"role": "user", "content": self._get_react_prompt(prompt)}

            for i in range(max_iterations):
                config = {"model": model_id, "temperature": 0.1}
                # Tool Attention (arXiv:2604.21816v1): inject only the tools
                # most relevant to the current turn instead of all schemas.
                active_tools = self._select_tools(tools) if tool_gating_enabled else tools
                msg = conversation.complete(tools=active_tools, config=config)
                conversation += msg
                if self.is_complete:
                    # According to the protocol, after disclose_reasoning with status="DONE",
                    # the agent should output a final assistant message with the complete answer.
                    # Check if this message has content (the final answer)
                    if msg.get("content"):
                        self.result = msg.get("content")
                        break
                    # If no content yet, allow one more iteration to get the final message
                    # (the tool call processing might need another round)
                    if i < max_iterations - 1:
                        continue
                    break

            # If we didn't capture the result yet, try to get it from the last message
            if not self.result:
                self.result = msg.get("content", "")
            self.outcome = "success"
        except BaseException as e:
            self.outcome = "error"
            raise e
