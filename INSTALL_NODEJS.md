# Installing Node.js for Windows

## Option 1: Download from Official Website (Recommended)

1. **Visit Node.js website:**
   - Go to: https://nodejs.org/
   - Download the **LTS (Long Term Support)** version
   - Choose the Windows Installer (.msi) for your system (64-bit recommended)

2. **Run the Installer:**
   - Double-click the downloaded `.msi` file
   - Follow the installation wizard
   - ✅ **Important:** Check "Add to PATH" option (usually checked by default)
   - Click "Install"

3. **Verify Installation:**
   - Open a **NEW** PowerShell/Command Prompt window
   - Run:
     ```powershell
     node --version
     npm --version
     ```
   - You should see version numbers (e.g., `v20.10.0` and `10.2.3`)

## Option 2: Using Chocolatey (If you have it)

```powershell
choco install nodejs
```

## Option 3: Using Winget (Windows Package Manager)

```powershell
winget install OpenJS.NodeJS.LTS
```

## After Installation

1. **Close and reopen your terminal** (PowerShell/Command Prompt)
   - This is important so PATH changes take effect

2. **Verify installation:**
   ```powershell
   node --version
   npm --version
   ```

3. **Navigate to frontend and install dependencies:**
   ```powershell
   cd Multilingual-Model\frontend
   npm install
   ```

## Troubleshooting

### If `npm` still not found after installation:

1. **Restart your computer** (sometimes needed for PATH to update)

2. **Check PATH manually:**
   - Open System Properties → Environment Variables
   - Check if `C:\Program Files\nodejs\` is in PATH
   - If not, add it manually

3. **Verify Node.js installation location:**
   - Usually: `C:\Program Files\nodejs\`
   - Check if `node.exe` and `npm.cmd` exist there

## Quick Check Commands

After installation, run these in a NEW terminal:

```powershell
# Check Node.js version
node --version

# Check npm version  
npm --version

# Check installation location
where.exe node
where.exe npm
```

## Next Steps

Once Node.js is installed:

1. ✅ Verify with `node --version` and `npm --version`
2. ✅ Navigate to frontend: `cd Multilingual-Model\frontend`
3. ✅ Install dependencies: `npm install`
4. ✅ Start dev server: `npm run dev`
