; Installing as 寻寻 over an install still called 企业知识库 (packaging/rename/README.md).
;
; A new product name means a new install folder and a new uninstall entry, so without this the
; old app would stay beside the new one. The identifier, the binary name and %APPDATA% stay the
; same, so conversations, data and the key carry over. The old uninstaller runs silently: it
; only deletes app data when its checkbox is ticked, and it removes shortcuts only if they
; point at its own folder.

!define OLD_PRODUCT "企业知识库"
!define OLD_UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${OLD_PRODUCT}"
!define OLD_LOCATIONKEY "Software\Enterprise KB\${OLD_PRODUCT}"

Var OldInstallRemoved

!macro NSIS_HOOK_PREINSTALL
  StrCpy $OldInstallRemoved 0
  ReadRegStr $R8 HKCU "${OLD_LOCATIONKEY}" ""
  ReadRegStr $R7 HKCU "${OLD_UNINSTKEY}" "UninstallString"
  ${If} $R7 != ""
  ${AndIf} $R8 != ""
  ${AndIf} ${FileExists} "$R8\uninstall.exe"
    DetailPrint "Removing the old install at $R8"
    ; _?= makes the uninstaller run in place, so this waits until it is done.
    ExecWait '"$R8\uninstall.exe" /S _?=$R8' $R9
    Delete "$R8\uninstall.exe"
    RMDir "$R8"
    DeleteRegKey HKCU "${OLD_LOCATIONKEY}"
    DeleteRegKey /ifempty HKCU "Software\Enterprise KB"
    StrCpy $OldInstallRemoved 1
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; The old shortcuts are gone, and an update (/UPDATE) normally leaves shortcuts alone.
  ${If} $OldInstallRemoved = 1
  ${AndIf} $UpdateMode = 1
    CreateShortcut "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    !insertmacro SetLnkAppUserModelId "$SMPROGRAMS\${PRODUCTNAME}.lnk"
    CreateShortcut "$DESKTOP\${PRODUCTNAME}.lnk" "$INSTDIR\${MAINBINARYNAME}.exe"
    !insertmacro SetLnkAppUserModelId "$DESKTOP\${PRODUCTNAME}.lnk"
  ${EndIf}
!macroend
