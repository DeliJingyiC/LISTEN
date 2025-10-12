# GitHub Upload Guide for LISTEN

Follow these steps to upload your LISTEN repository to GitHub.

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click the `+` icon in the top right → "New repository"
3. Fill in:
   - **Repository name**: `LISTEN`
   - **Description**: "LISTEN: Measuring Lexical vs. Acoustic Emotion Cues Reliance in Audio LLMs"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

## Step 2: Configure Git (First Time Only)

```bash
cd /users/PAS2062/delijingyic/project/LISTEN

# Set your git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Add Files to Git

```bash
cd /users/PAS2062/delijingyic/project/LISTEN

# Check what files will be committed
git status

# Add all files
git add .

# Check what's staged
git status

# If you see any large files or sensitive data, remove them:
# git reset HEAD path/to/large/file
```

## Step 4: Create Your First Commit

```bash
# Commit your changes
git commit -m "Initial commit: LISTEN benchmark with baseline models and evaluation scripts"
```

## Step 5: Connect to GitHub

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/LISTEN.git

# Verify the remote was added
git remote -v
```

## Step 6: Push to GitHub

```bash
# Push your code
git branch -M main
git push -u origin main
```

If this is your first time pushing, GitHub will ask you to authenticate:
- **Option 1**: Use a Personal Access Token (recommended)
  1. Go to GitHub → Settings → Developer settings → Personal access tokens
  2. Generate new token with `repo` scope
  3. Use this token as your password when prompted
  
- **Option 2**: Use SSH keys
  1. Follow [GitHub's SSH guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

## Step 7: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/LISTEN`
2. You should see all your files
3. The README.md will be displayed on the main page

## Optional: Add Large Files with Git LFS

If you have large model files or datasets:

```bash
# Install Git LFS
git lfs install

# Track large files (e.g., model checkpoints)
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

## Common Issues

### Issue 1: Files Too Large
**Error**: `remote: error: File X is 100.00 MB; this exceeds GitHub's file size limit`

**Solution**: 
```bash
# Remove the large file from git
git rm --cached path/to/large/file

# Add it to .gitignore
echo "path/to/large/file" >> .gitignore

# Commit and push
git add .gitignore
git commit -m "Remove large file from tracking"
git push
```

### Issue 2: Authentication Failed
**Solution**: Use a Personal Access Token instead of password

### Issue 3: Nothing to Commit
**Solution**: Make sure files aren't in .gitignore

```bash
# Check git status
git status

# If files are missing, check .gitignore
cat .gitignore
```

## Future Updates

After the initial upload, to push new changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with a descriptive message
git commit -m "Your commit message here"

# Push to GitHub
git push
```

## Tips

1. **Commit often**: Make small, focused commits with clear messages
2. **Don't commit sensitive data**: API keys, passwords, tokens
3. **Use .gitignore**: Exclude large files, temporary files, and generated files
4. **Write good commit messages**: Explain what and why, not just what changed

## Need Help?

- [GitHub Documentation](https://docs.github.com)
- [Git Documentation](https://git-scm.com/doc)
- Open an issue on your repository for community help

