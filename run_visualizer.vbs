Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Check if venv exists
venvPath = scriptDir & "\venv_geo_sense"
If Not objFSO.FolderExists(venvPath) Then
    MsgBox "ERROR: Virtual environment 'venv_geo_sense' not found." & vbCrLf & "Please run install.bat first.", vbCritical, "GeoSense"
    WScript.Quit
End If

' Run the application using pythonw from the venv (hidden)
pythonwPath = venvPath & "\Scripts\pythonw.exe"
appPath = scriptDir & "\seismic_app.py"
objShell.Run """" & pythonwPath & """ """ & appPath & """", 0, False
