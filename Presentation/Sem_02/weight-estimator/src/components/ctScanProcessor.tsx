"use client";

import {useState, useRef, useEffect} from "react";
import {Play, Weight} from "lucide-react";
import {motion} from "framer-motion";
import Dropdown from "./ui/dropdown";
import Model from "./ui/model";
import {ArcherContainer, ArcherElement} from "react-archer";
import {distriCalculatro} from "./utils/distributionCalculator";
//import {useCtScanImage} from "./utils/ctScanProcessing";
import Image from "next/image";

//const ctScans = ["Brain CT", "Chest CT", "Abdominal CT", "Spine CT"];
const models = [
  "Best Segmentation",
  "Best Histogram",
  "Best 2D",
  "Best 3D",
  "Baseline",
];

const ctScans = [
  {
    name: "Scan 1",
    url: "/ct_scans/Scan1.png",
    weight: 80.0,
    prediction: [81.68, 82.78, 81.51, 79.74, 74.75],
  },
  {
    name: "Scan 2",
    url: "/ct_scans/Scan2.png",
    weight: 67.0,
    prediction: [64.87, 64.36, 61.88, 62.41, 70.0],
  },
  {
    name: "Scan 3",
    url: "/ct_scans/Scan3.png",
    weight: 85.0,
    prediction: [82.67, 80.43, 87.38, 84.44, 77.0],
  },
  {
    name: "Scan 4",
    url: "/ct_scans/Scan4.png",
    weight: 92.0,
    prediction: [84.25, 86.83, 93.46, 92.58, 70.0],
  },
  {
    name: "Scan 5",
    url: "/ct_scans/Scan5.png",
    weight: 45.0,
    prediction: [45.29, 44.42, 44.02, 57.2, 71.33],
  },
  {
    name: "Scan 6",
    url: "/ct_scans/Scan6.png",
    weight: 110.0,
    prediction: [105.53, 107.11, 113.05, 98.09, 90.0],
  },
  {
    name: "Scan 7",
    url: "/ct_scans/Scan7.png",
    weight: 63.0,
    prediction: [66.25, 65.47, 73.85, 64.37, 55.67],
  },
  {
    name: "Scan 8",
    url: "/ct_scans/Scan8.png",
    weight: 68.0,
    prediction: [61.27, 62.18, 65.15, 64.5, 57.67],
  },
  {
    name: "Scan 9",
    url: "/ct_scans/Scan9.png",
    weight: 79.0,
    prediction: [80.76, 79.93, 76.79, 79.26, 72.0],
  },
];

export default function CTScanProcessor() {
  const [selectedScan, setSelectedScan] = useState(ctScans[0]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState([
    "...kg",
    "...kg",
    "...kg",
    "...kg",
    "...kg",
  ]);

  const [original, setOriginal] = useState("...kg");

  const startProcessing = () => {
    setIsProcessing(true);
    setOriginal("...kg");
    setResults(["...kg", "...kg", "...kg", "...kg", "...kg"]);
    setTimeout(() => {
      setResults(
        selectedScan.prediction.map(
          prediction =>
            prediction +
            "kg          " +
            (selectedScan.weight - prediction > 0 ? "" : "+") +
            (prediction - selectedScan.weight).toFixed(2) +
            "kg",
        ),
      );
      setIsProcessing(false);
      setOriginal(selectedScan.weight.toString() + "kg");
    }, 3000);
  };

  return (
    <div
      className="min-h-screen bg-cover bg-center flex items-center justify-center"
      style={{
        backgroundImage:
          "url('https://images.unsplash.com/photo-1557683311-eac922347aa1?auto=format&fit=crop&q=80&w=2029&ixlib=rb-4.0.3')",
      }}>
      <div className="bg-black bg-opacity-50 p-8 rounded-lg w-full">
        <ArcherContainer strokeColor="red">
          <div className="flex">
            {/* First Column: Dropdown and Start Button */}
            <div className="w-1/3 pr-4">
              <Dropdown
                options={ctScans.map(scan => scan.name)}
                selected={selectedScan.name}
                onSelect={name => {
                  const scan = ctScans.find(scan => scan.name === name);
                  if (scan) setSelectedScan(scan);
                }}
              />
              <ArcherElement id="start-button">
                <button
                  id="start-button"
                  className="w-full bg-blue-500 text-white p-2 rounded flex items-center justify-center mt-4"
                  onClick={startProcessing}
                  disabled={isProcessing}>
                  <Play className="mr-2" /> Start
                </button>
              </ArcherElement>

              <div className="border rounded-lg p-4 bg-gray-50 mt-4">
                <Image
                  src={selectedScan.url}
                  alt={`${selectedScan.name} visualization`}
                  width={500}
                  height={500}
                  className="w-full h-auto"
                />
              </div>
            </div>

            {/* Second Column: Models */}
            <div className="w-1/3 relative">
              <div className="flex flex-col space-y-4">
                {models.map(name => (
                  <Model name={name} isProcessing={isProcessing} />
                ))}
              </div>
            </div>

            {/* Third Column: Results */}
            <div className="w-1/3 pl-4">
              <div className="bg-white rounded p-4">
                <h2 className="text-xl font-bold mb-4 text-black">Results</h2>
                <div className="mb-2">
                  <span className="font-semibold text-black">Original:</span>
                  <span className="text-green-500 font-bold ml-2">
                    {original}
                  </span>
                </div>
                {models.map((name, index) => (
                  <div key={index} className="mb-2">
                    <span className="font-semibold text-black">{name}:</span>
                    <span className="text-blue-500 font-bold ml-2">
                      {results[index]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </ArcherContainer>
      </div>
    </div>
  );
}
