"""Session context built across agent loop turns."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.types import TurnRecord


class SessionContext(BaseModel):
    """Accumulated state for one user task."""

    user_task: str
    turns: list[TurnRecord] = Field(default_factory=list)
    user_replies: list[str] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)

    def add_turn(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def add_parse_error(self, error: str) -> None:
        self.parse_errors.append(error)

    def add_user_reply(self, reply: str) -> None:
        self.user_replies.append(reply)

    def to_prompt(self) -> str:
        """Serialize full context for the model's user message."""
        sections: list[str] = [
            "## User task",
            self.user_task.strip(),
        ]

        if self.user_replies:
            sections.append("\n## User clarifications")
            for i, reply in enumerate(self.user_replies, start=1):
                sections.append(f"{i}. {reply}")

        if self.turns:
            sections.append("\n## Prior turns")
            for record in self.turns:
                sections.append(self._format_turn(record))

        if self.parse_errors:
            sections.append("\n## Parse errors (fix your JSON)")
            for err in self.parse_errors:
                sections.append(f"- {err}")

        sections.append(
            "\n## Your turn\n"
            "Return the next JSON action object to continue or finish the task."
        )
        return "\n".join(sections)

    @staticmethod
    def _format_turn(record: TurnRecord) -> str:
        step = record.step
        lines = [f"\n### Turn {record.turn}"]
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        lines.append(f"Action: {step.action.value}")

        if record.parse_error:
            lines.append(f"Parse note: {record.parse_error}")

        if step.command:
            lines.append(f"Command: {step.command}")
        if step.code:
            lines.append(f"Code:\n```python\n{step.code}\n```")
        if step.message:
            lines.append(f"Message: {step.message}")

        if record.tool_result is not None:
            tr = record.tool_result
            lines.append(f"Executed: {tr.executed}")
            lines.append(f"Exit code: {tr.exit_code}")
            lines.append(f"Timed out: {tr.timed_out}")
            if tr.stdout:
                lines.append(f"Stdout:\n{tr.stdout}")
            if tr.stderr:
                lines.append(f"Stderr:\n{tr.stderr}")

        return "\n".join(lines)
