import {useState, useEffect} from "react";

// This function would typically be on the server-side
// For this example, we're simulating it on the client
function createDummyCtScan(width: number, height: number): number[][] {
  const array = [];
  for (let i = 0; i < height; i++) {
    const row = [];
    for (let j = 0; j < width; j++) {
      // Create a simple pattern
      row.push((i + j) % 256);
    }
    array.push(row);
  }
  return array;
}

export function useCtScanImage(scanType: string) {
  const [imageData, setImageData] = useState<string | null>(null);

  useEffect(() => {
    // In a real application, this would fetch data from an API
    const ctScanArray = createDummyCtScan(256, 256);

    // Convert the 2D array to a flat Uint8ClampedArray with RGBA values
    const flatArray = new Uint8ClampedArray(256 * 256 * 4);
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        const value = ctScanArray[i][j];
        const index = (i * 256 + j) * 4;
        flatArray[index] = value; // R
        flatArray[index + 1] = value; // G
        flatArray[index + 2] = value; // B
        flatArray[index + 3] = 255; // A (fully opaque)
      }
    }

    // Create ImageData
    const imageData = new ImageData(flatArray, 256, 256);

    // Create a temporary canvas to draw the image data
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.putImageData(imageData, 0, 0);
      setImageData(canvas.toDataURL());
    }
  }, [scanType]);

  return imageData;
}
