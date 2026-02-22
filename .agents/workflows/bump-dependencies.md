---
description: how to bump dependencies
---

This workflow guides you through updating the site's dependencies to their latest compatible versions.

1. **Check for updates**
   Run the following command to see which gems can be updated:
   ```bash
   bundle outdated
   ```

2. **Update dependencies**
   // turbo
   Update all gems to the latest allowed versions:
   ```bash
   bundle update
   ```

3. **Verify the build**
   // turbo
   Ensure the site still builds correctly and passes tests:
   ```bash
   bash tools/test.sh
   ```

4. **Commit changes**
   If everything looks good, commit the updated `Gemfile.lock`.
