"""Strict tool JSON decoding with bounded, content-free repair diagnostics."""
import json

from fg_agents.core.types import StopReason, ToolArgumentDiagnostic, ToolCall


def decode_tool_call(tool_id: str, name: str, fragments: list[str],
                     stop: StopReason, *, complete: bool = True) -> ToolCall:
    raw = ''.join(fragments)
    code = None
    location = {}
    if not complete:
        code = 'incomplete_stream'
    elif not raw and stop == StopReason.MAX_TOKENS:
        code = 'missing_arguments'
    else:
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            code = 'invalid_json'
            location = {'line': error.lineno, 'column': error.colno, 'position': error.pos}
        else:
            if isinstance(arguments, dict):
                return ToolCall(id=tool_id, tool_name=name, arguments=arguments)
            code = 'non_object'
    diagnostic = ToolArgumentDiagnostic(code=code, characters=len(raw),
                                        fragments=len(fragments), stop_reason=stop, **location)
    reason = ('Tool arguments were cut off by the output token limit. Re-issue a smaller complete JSON object.'
              if stop == StopReason.MAX_TOKENS else
              'Tool arguments were not a valid JSON object. Re-issue this call with valid JSON.')
    where = f" line {location['line']}, column {location['column']};" if location else ''
    # Do not use JSONDecodeError.msg/doc or echo arbitrary provider content.
    detail = f' [{code};{where} {len(raw)} chars; {len(fragments)} fragments; stop={stop.value}]'
    return ToolCall(id=tool_id, tool_name=name, arguments_error=reason + detail,
                    arguments_diagnostic=diagnostic)
