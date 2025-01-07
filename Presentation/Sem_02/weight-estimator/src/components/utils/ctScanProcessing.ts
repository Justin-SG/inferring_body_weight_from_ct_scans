import {useState, useEffect} from "react";
import {NumpyLoader} from "numpyjs";

export async function loadNpyFile(url: string): Promise<number[][]> {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const npLoader = new NumpyLoader();
  const npArray = await npLoader.load(new Uint8Array(arrayBuffer));
  return npArray.data as number[][];
}

export function normalizeData(data: number[][]): number[][] {
  const flattened = data.flat();
  const min = Math.min(...flattened);
  const max = Math.max(...flattened);
  return data.map(row =>
    row.map(val => Math.floor(((val - min) / (max - min)) * 255)),
  );
}

export function useCtScanImage(scanUrl: string) {
  const [imageData, setImageData] = useState<string | null>(null);

  useEffect(() => {
    async function loadAndProcessImage() {
      try {
        const npArray = await loadNpyFile(scanUrl);
        const normalizedData = normalizeData(npArray);

        const canvas = document.createElement("canvas");
        canvas.width = normalizedData[0].length;
        canvas.height = normalizedData.length;
        const ctx = canvas.getContext("2d");

        if (ctx) {
          const imageData = ctx.createImageData(canvas.width, canvas.height);
          for (let i = 0; i < normalizedData.length; i++) {
            for (let j = 0; j < normalizedData[i].length; j++) {
              const index = (i * normalizedData[i].length + j) * 4;
              const value = normalizedData[i][j];
              imageData.data[index] = value; // R
              imageData.data[index + 1] = value; // G
              imageData.data[index + 2] = value; // B
              imageData.data[index + 3] = 255; // A (fully opaque)
            }
          }
          ctx.putImageData(imageData, 0, 0);
          setImageData(canvas.toDataURL());
        }
      } catch (error) {
        console.error("Error loading or processing CT scan:", error);
        setImageData(null);
      }
    }

    loadAndProcessImage();
  }, [scanUrl]);

  return imageData;
}
