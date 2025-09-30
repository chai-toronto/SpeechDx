# this should be a .sh file that pulls the cambridge data from the official source

source ./download/config.sh

DEST="$ROOT/Cambridge"
mkdir -p "$DEST/raw"
mkdir -p "$DEST/processed"

# wget  ...