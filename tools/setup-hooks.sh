#!/usr/bin/env bash
#
# Install git hooks for the repository
#

set -eu

HOOK_DIR=".git/hooks"
PRE_PUSH_HOOK="$HOOK_DIR/pre-push"

if [ ! -d ".git" ]; then
    echo "Error: .git directory not found. Please run this from the repository root."
    exit 1
fi

echo "Installing pre-push hook..."

cat <<'EOF' > "$PRE_PUSH_HOOK"
#!/usr/bin/env bash
# Git pre-push hook to enforce local tests before allowing a push to remote.

# Skip if we are pushing to a different remote or if it's a delete operation
# (Optional: specialized logic for branch filtering could go here)

echo ""
echo "🚀 Enforcing local tests before push..."
echo "--------------------------------------------------"

# Run the local test suite
# This performs a production build and runs htmlproofer
bash tools/test.sh

RESULT=$?

echo "--------------------------------------------------"

if [ $RESULT -ne 0 ]; then
    echo "❌ Tests failed! Push aborted."
    echo ""
    echo "Please fix the issues reported above."
    echo "Tip: You can use 'git push --no-verify' to skip this check in emergencies."
    echo ""
    exit 1
fi

echo "✅ All tests passed. Proceeding with push..."
echo ""
exit 0
EOF

chmod +x "$PRE_PUSH_HOOK"

echo "Success: Git pre-push hook installed at $PRE_PUSH_HOOK"
