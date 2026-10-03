# Launches run_model_test.py under XWayland so Open3D/Qt work on Wayland.
# Any extra arguments are passed through to the Python script.

# Run from the directory this script lives in, so it works from anywhere
cd "$(dirname "$(readlink -f "$0")")" || exit 1

export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb

exec python run_model_test.py "$@"