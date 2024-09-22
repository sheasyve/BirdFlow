## Extensions can be built, validated & installed via command-line.

#### To build the package defined in the current directory use the following commands:
``` bash
blender --command extension build
```
#### See build docs.

—

## To validate the manifest without building the package:
``` bash
blender --command extension validate
```
## You may also validate a package without having to extract it first.
```bash
blender --command extension validate add-on-package.zip
```
# Blender CLI
### https://docs.blender.org/manual/en/latest/advanced/command_line/extension_arguments.html#command-line-args-extensions
## Install
```bash
blender --command extension install [-h] [-s] [-e] [--no-prefs]                        PACKAGES
```
positional arguments:
PACKAGES:
The packages to operate on (separated by , without spaces).

options:
-h, --help
show this help message and exit

-s, --sync
Sync the remote directory before performing the action.

-e, --enable
Enable the extension after installation.

--no-prefs
Treat the user-preferences as read-only, preventing updates for operations that would otherwise modify them. This means removing extensions or repositories for example, wont update the user-preferences.

## Uninstall
```bash 
blender --command extension remove [-h] [--no-prefs] PACKAGES
```
Disable & remove package(s).

positional arguments:
PACKAGES:
The packages to operate on (separated by , without spaces).

options:
-h, --help
show this help message and exit

--no-prefs
Treat the user-preferences as read-only, preventing updates for operations that would otherwise modify them. This means removing extensions or repositories for example, wont update the user-preferences.