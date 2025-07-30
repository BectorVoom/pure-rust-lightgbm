/code-check
# Code Check Procedure

1. **Run Compiler Check**
   ```bash
   cargo check
   ```

2. **Consolidate Warnings and Errors**
   - Review and document every warning and error reported.

3. **Create TODOs**
   - For each warning and error, draft a clear TODO item describing how it should be resolved.

4. **Open GitHub Issues**
   - Convert each TODO into a GitHub Issue against the `<owner>/<project-name>` repository.

5. **Verify Issue Creation**
   ```bash
   gh issue list --repo <owner>/<project-name> --state open --json number,title
   ```

6. **Report**
   - Summarize the results—list of created issues and their identifiers—and share with the team.
```
