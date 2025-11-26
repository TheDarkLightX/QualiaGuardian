"""
Compatibility wrapper package for legacy imports expecting
`guardian_ai_tool.guardian`. The modern codebase lives in the
`guardian` package directly; tests that still import through the old
namespace can do so without modification thanks to this shim.
"""
