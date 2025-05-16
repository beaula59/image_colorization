#!/bin/bash

INPUT_FILE="synset_ids.txt"

while read -r SYNSET CLASS_NAME; do
    echo "Processing $SYNSET ($CLASS_NAME)..."

    if [ -d "$CLASS_NAME" ]; then
        JPEG_COUNT=$(find "$CLASS_NAME" -type f \( -iname "*.jpg" -o -iname "*.JPEG" \) | wc -l)

        if [ "$JPEG_COUNT" -gt 1000 ]; then
            echo "Directory $CLASS_NAME exists and has $JPEG_COUNT JPEG images. Skipping..."
            continue
        else
            echo "Directory $CLASS_NAME exists but has only $JPEG_COUNT JPEG images. Deleting..."
            rm -rf "$CLASS_NAME"
        fi
    fi

    # Download the tar file
    wget -c "https://image-net.org/data/winter21_whole/${SYNSET}.tar" -O "${SYNSET}.tar"

    # Create a folder with class name
    mkdir -p "$CLASS_NAME"

    # Extract into the folder
    tar -xf "${SYNSET}.tar" -C "$CLASS_NAME"

    # Remove the tar file
    rm "${SYNSET}.tar"

    echo "Finished $CLASS_NAME"
done < "$INPUT_FILE"

