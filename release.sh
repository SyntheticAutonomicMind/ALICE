#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)
#
# ALICE Release Script
#
# Automates the release process:
# 1. Validate version format (YYYYMMDD.RELEASE)
# 2. Update src/__init__.py with new version
# 3. Commit the version change
# 4. Create and push annotated tag
# 5. GitHub Actions workflow handles the rest:
#    - Builds distribution tarball
#    - Creates GitHub Release with assets
#    - Builds and pushes Docker images (CPU, CUDA, ROCm)
#
# Usage:
#   ./release.sh <version>
#
# Examples:
#   ./release.sh 20260222.2
#   ./release.sh 20260223.1

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Check argument
if [ $# -ne 1 ]; then
    echo -e "${RED}Usage: $0 <version>${NC}"
    echo "Example: $0 20260222.2"
    echo ""
    echo "Version format: YYYYMMDD.RELEASE (e.g., 20260222.1)"
    exit 1
fi

VERSION="$1"

# Validate version format (YYYYMMDD.N)
if ! echo "$VERSION" | grep -qE '^[0-9]{8}\.[0-9]+$'; then
    echo -e "${RED}ERROR: Invalid version format: ${VERSION}${NC}"
    echo "Expected format: YYYYMMDD.RELEASE (e.g., 20260222.1)"
    exit 1
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  ALICE Release Script${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Preparing release: ${CYAN}${VERSION}${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Not in a git repository${NC}"
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}ERROR: You have uncommitted changes${NC}"
    echo "Please commit or stash them before creating a release."
    git status --short
    exit 1
fi

# Check if tag already exists
if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Tag v${VERSION} already exists${NC}"
    echo "To delete it: git tag -d v${VERSION} && git push origin :refs/tags/v${VERSION}"
    exit 1
fi

# Get current version
CURRENT=$(python3 -c "
with open('src/__init__.py') as f:
    for line in f:
        if '__version__' in line:
            print(line.split('\"')[1])
            break
" 2>/dev/null || echo "unknown")

echo -e "  Current version: ${YELLOW}${CURRENT}${NC}"
echo -e "  New version:     ${GREEN}${VERSION}${NC}"
echo ""

# Step 1: Update src/__init__.py
echo -e "${CYAN}Step 1:${NC} Updating src/__init__.py..."
sed -i.bak "s/__version__ = \".*\"/__version__ = \"${VERSION}\"/" src/__init__.py
rm -f src/__init__.py.bak

# Verify update
UPDATED=$(python3 -c "
with open('src/__init__.py') as f:
    for line in f:
        if '__version__' in line:
            print(line.split('\"')[1])
            break
" 2>/dev/null)

if [ "$UPDATED" != "$VERSION" ]; then
    echo -e "${RED}ERROR: Version update failed${NC}"
    echo "Expected: $VERSION, Got: $UPDATED"
    git checkout src/__init__.py
    exit 1
fi
echo -e "  ${GREEN}✓${NC} src/__init__.py updated to ${VERSION}"

# Step 2: Commit
echo -e "${CYAN}Step 2:${NC} Committing version update..."
git add src/__init__.py
git commit -m "chore(release): prepare version ${VERSION}"
echo -e "  ${GREEN}✓${NC} Committed"

# Step 3: Create annotated tag
echo -e "${CYAN}Step 3:${NC} Creating tag v${VERSION}..."
git tag -a "v${VERSION}" -m "ALICE ${VERSION}"
echo -e "  ${GREEN}✓${NC} Tag v${VERSION} created"

# Step 4: Push
echo -e "${CYAN}Step 4:${NC} Pushing to origin..."
git push origin main
git push origin "v${VERSION}"
echo -e "  ${GREEN}✓${NC} Pushed to origin"

# Summary
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Release v${VERSION} published!${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  GitHub Actions is now:"
echo "    1. Building distribution tarball (alice-${VERSION}.tar.gz)"
echo "    2. Creating GitHub Release"
echo "    3. Building Docker images (CPU, CUDA, ROCm)"
echo ""
echo "  Monitor: https://github.com/SyntheticAutonomicMind/ALICE/actions"
echo "  Release: https://github.com/SyntheticAutonomicMind/ALICE/releases/tag/v${VERSION}"
echo ""
echo "  Docker images will be available at:"
echo "    ghcr.io/syntheticautonomicmind/alice:${VERSION}"
echo "    ghcr.io/syntheticautonomicmind/alice:${VERSION}-cuda"
echo "    ghcr.io/syntheticautonomicmind/alice:${VERSION}-rocm"
echo ""
echo "  Running instances will detect this update within 1 hour"
echo "  (or immediately via POST /v1/update/check)"
echo ""
