# Streaming tool arguments

Each tool index owns its original argument text, display preview, and incremental
top-level path scanner. Original text is retained for final decoding and errors.

Display previews preserve nested objects and arrays, and may expose unfinished
strings or numeric prefixes. They are not execution or security inputs. Display
reparsing uses a growth threshold of `max(256, current_length // 32)` characters.
The independent path scanner consumes every delta, so display throttling cannot
delay a completed top-level path check. Nested paths and unfinished strings are
not exposed to that check. Early rejection aborts the model turn; final runtime
policies still run for accepted calls.

Final streamed and non-streamed arguments share `parse_final_arguments`.
Standard JSON is attempted first; the fallback supports single-quoted strings
and trailing commas. Unlike display previews, final decoding never auto-closes
containers or strings. Duplicate keys, trailing content, non-object arguments,
and non-finite numbers are rejected. This is intentionally narrower than Oh My
Pi's relaxed grammar; unquoted keys, comments and Python literals are not repaired.

`toolcall_end` means collection ended, not necessarily successful parsing.
`arguments_error` exposes a parse failure to event consumers. The original call
is retained with `argument_error` for dispatch, where it becomes an error result
without invoking the runner. Valid sibling calls remain eligible for normal
policy checks and scheduling. This does not override turn-level truncation or
early security rejection.
