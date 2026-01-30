# Instructions for Claude

See project_plan.md for the full project specification and phase1.md for the research foundation.

## Debugging Philosophy

When tests fail or bugs are discovered:

1. **Don't throw spaghetti at the wall** - No trial-and-error fixes
2. **Diagnose carefully first** - Understand the exact root cause before making changes
3. **Add type hints and debug output** to trace data flow when needed
4. **Fix the actual bug** - Don't work around it in tests
5. **Establish high confidence** - Only make changes when you meticulously understand what's wrong

## When Working on Issues

- Remove attempted fixes that didn't help after the real fix is found

## Never Deviate From Instructions Without Permission

- **NEVER implement a "simpler approach" or skip parts of instructions without explicit user approval**
- If something seems too complex to implement, STOP and ask - do not proceed with a partial implementation
- Before verifying a change, write a verification plan that tests both correctness and the original goal (e.g., if implementing a memory optimization, verify both that it works AND that memory actually decreased)
- When verification fails or times out, fix the verification approach and re-run - don't declare success based on partial results
- Never leave TODO comments indicating deferred compliance - either implement what was asked or ask for permission to defer
- If you're uncertain how to implement something, ask rather than guessing or simplifying

## Physical Simulation Integrity

When implementing or modifying the distillation column simulator:

1. **Never change thermodynamic parameters without explicit approval**
   - Default mixture: methanol-water with NRTL (α=0.1)
   - Antoine coefficients must match NIST sources
   - Hydraulic parameters (τ_L: 0.5-15s, j: -5 to +5) are calibrated values from literature

2. **Conservation laws must hold**
   - Mass balance closure < 0.1% error at all times
   - Energy balance closure < 1% error
   - Always verify both after any physics changes

3. **Numerical stability requirements**
   - No NaN/Inf during normal operation
   - Stable for 10,000+ timesteps
   - Composition bounds enforced: x,y ∈ [0,1], holdups ≥ 0

## JAX-Specific Requirements

- All core dynamics must be JIT-compilable and vmap-compatible
- Use pure functions with no side effects in the simulator core
- State representations use chex.dataclass or flax.struct
- Fixed-step integrators (RK4) for determinism and vectorization

## Validation Before Merging

- Antoine equations match NIST values within tolerance
- Step responses show correct direction (↑reflux → ↑purity, ↑reboiler duty → ↑vapor flow)
- Temperature profile is monotonic (increases down column at steady state)
- Flooding/weeping constraints trigger at appropriate thresholds
- Gymnasium API check passes

## Key References

See phase1.md for detailed equations and parameters. Primary sources:
- Wittgens & Skogestad (2000) for hydraulics and control-relevant dynamics
- NIST WebBook for Antoine coefficients
- NPTEL Module 7 for flooding/weeping correlations
- Armfield UOP3 specs for teaching column geometry

## To-Do List Management

- Always maintain a to-do list for multi-step tasks using the TodoWrite tool
- After completing each to-do item, print the full to-do list with status indicators:
  - Use `[x]` for completed items
  - Use `[ ]` for incomplete items
- IMPORTANT: Use a code block to prevent markdown from stripping the checkboxes
- Example format:

```
- [x] Completed task one
- [x] Completed task two
- [ ] Pending task three
- [ ] Pending task four
```

## Documentation Integrity

- **NEVER remove test results, validation criteria, or status items to make a document look better**
- If a test was not run, mark it as "Not tested" - do not delete the row
- If a test failed or is incomplete, report it honestly - do not hide it
- Omitting negative or incomplete results is unethical and misleading
- All claims in documentation must be backed by actual test runs

## Communication

- Always write at least a sentence to describe every action taken

## General Behavior

- Read files before modifying them
- For code that is intended to persist indefinitely, prefer refactoring for modularity instead of producing redundant code
- Use the TodoWrite tool to track multi-step tasks
- Mark todos as completed immediately after finishing them
- Always prefer editing existing files over creating new ones
- Reuse code when possible
- Before writing new code, search the codebase for existing implementations of the same functionality
- When running validation tests, always check both physical consistency (mass/energy balance) and behavioral correctness (step response directions)
- When running experiments or validations, document: (1) the context and goal, (2) detailed results, (3) discussion of what the results mean for sim-to-real transfer, and (4) recommended next steps

