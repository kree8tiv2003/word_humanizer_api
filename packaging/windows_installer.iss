; Inno Setup script: builds ScriptStudio-Setup-Windows.exe from the PyInstaller output in dist\Script Studio.
#define AppVersion GetEnv("APP_VERSION")
#if AppVersion == ""
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8E2B6C1A-5C77-4E0B-9C1E-6A1F3B0D2E51}
AppName=Script Studio
AppVersion={#AppVersion}
AppPublisher=Script Studio
DefaultDirName={localappdata}\Programs\Script Studio
DefaultGroupName=Script Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\installer
OutputBaseFilename=ScriptStudio-Setup-Windows
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\Script Studio.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

CloseApplications=force
RestartApplications=no

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  { Close a running Script Studio so every file is replaced by the new version. }
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /IM "Script Studio.exe"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(800);
  Result := '';
end;

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\Script Studio\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Script Studio"; Filename: "{app}\Script Studio.exe"
Name: "{autodesktop}\Script Studio"; Filename: "{app}\Script Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Script Studio.exe"; Description: "Open Script Studio now"; Flags: nowait postinstall skipifsilent
