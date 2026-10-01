# Installing Script Studio on your computer

There's nothing to set up by hand. Download one file, open it, and paste your key once.

## 1. Download

Go to the **Releases** page of this repository and open **Script Studio** (marked *script-studio-latest*). Under **Assets**, click the file for your computer:

| Your computer | File to download |
|---|---|
| Windows 10 or 11 | `ScriptStudio-Setup-Windows.exe` |
| Mac with an Apple chip (M1, M2, M3, M4) | `ScriptStudio-Mac-AppleSilicon.dmg` |
| Older Mac with an Intel chip | `ScriptStudio-Mac-Intel.dmg` |

*Not sure which Mac you have?* Click the Apple menu → **About This Mac**. "Chip: Apple M…" means Apple chip; "Processor: Intel" means Intel.

The download is about 250–400 MB.

## 2. Install

### Windows
1. Double-click **ScriptStudio-Setup-Windows.exe** in your Downloads folder.
2. If a blue box says *"Windows protected your PC"*, click **More info**, then **Run anyway**. Windows shows this for new apps that aren't from the Microsoft Store.
3. Click **Next** through the installer and tick **Create a desktop shortcut**.
4. Click **Finish**. Script Studio opens in your web browser.

From now on, open it from the **Script Studio** icon on your desktop or in the Start menu.

### Mac
1. Double-click the **.dmg** file in your Downloads folder.
2. Drag **Script Studio** onto the **Applications** folder in the window that opens.
3. Open your **Applications** folder. **Hold the Control key, click Script Studio, and choose Open**, then click **Open** again. You only need to do this the first time.
   - If the box only offers *Move to Trash* or *Done*, open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to Script Studio.
4. Script Studio opens in your web browser.

## 3. First time: add your key

Script Studio uses Claude, from Anthropic, to write the scripts, so it needs your own Anthropic API key. The Settings window opens by itself the first time:

1. Click the **console.anthropic.com** link and sign in, or create an account.
2. Add some credit under **Billing**. Each script costs a few cents to a few dollars, depending on its length.
3. Go to **API Keys** → **Create Key**, give it any name, and click **Copy**.
4. Back in Script Studio, paste the key and click **Save**. You'll see **✓ Your Anthropic key works.**

The key is saved only on your computer. You can change it later with the **⚙ Settings** button.

## 4. Using it

- **Story → Script:** drop in a Word file, PDF, text file or audio recording, or paste a link or text.
- **Music Video:** drop in the song. You can also add lyrics or a script.
- Pick a length: **5 s, 10 s, 15 s, 30 s, 60 s** or **Full script**. Choose a style, then click **Write the script**.
- Segments appear as they're written. Use **Copy prompt** on any shot, **Edit** to change text, or **Rewrite…** to redo a segment with a note.
- **Export** gives you a PDF screenplay, a Fountain screenplay file, a shot list spreadsheet (CSV), a prompts-only file, Markdown or JSON.

A small **"Script Studio is running"** window stays open while you work. To reopen the browser tab, click **Open Script Studio** in that window. To quit, click **Quit** or close the window. Your scripts are saved, and they're listed under **Recent scripts** next time.

**Audio stories:** speech is turned into text on your computer for free. The first time you upload audio, the app downloads a speech model (about 150 MB), so give it a minute. If you have an OpenAI account, you can paste an OpenAI key in Settings for faster, more accurate transcription.

## Troubleshooting

| Problem | What to do |
|---|---|
| The browser didn't open | Click **Open Script Studio** in the small Script Studio window. |
| "No Anthropic API key yet" | Click **⚙ Settings** and paste your key. |
| "That key was rejected" | Copy the key again from console.anthropic.com and make sure you got all of it. |
| The script stops partway | Click **Resume**. Everything already written is kept. |
| An audio file won't load | Save or convert it as MP3 or WAV and try again. |
| Windows Firewall asks about Script Studio | Click **Allow**. It only talks to your own computer and to Anthropic. |

Your scripts and settings are kept in:
- **Windows:** `%APPDATA%\ScriptStudio`
- **Mac:** `~/Library/Application Support/ScriptStudio`

Uninstalling the app leaves this folder, so your scripts are safe.
