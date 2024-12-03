"use client";

import {useState, useRef, useEffect} from "react";
import {Play} from "lucide-react";
import {motion} from "framer-motion";
import Dropdown from "./ui/dropdown";
import Model from "./ui/model";
import {ArcherContainer, ArcherElement} from "react-archer";

const ctScans = ["Brain CT", "Chest CT", "Abdominal CT", "Spine CT"];

export default function CTScanProcessor() {
  const [selectedScan, setSelectedScan] = useState(ctScans[0]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(["", "", "", ""]);

  const startProcessing = () => {
    setIsProcessing(true);
    setTimeout(() => {
      setResults(["Result 1", "Result 2", "Result 3", "Result 4"]);
      setIsProcessing(false);
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
                options={ctScans}
                selected={selectedScan}
                onSelect={setSelectedScan}
              />
              <ArcherElement
                id="start-button"
                relations={[0, 1, 2, 3].map(index => ({
                  targetId: `model-${index}`,
                  targetAnchor: "left",
                  sourceAnchor: "right",
                  style: {strokeWidth: 2},
                }))}>
                <button
                  id="start-button"
                  className="w-full bg-blue-500 text-white p-2 rounded flex items-center justify-center mt-4"
                  onClick={startProcessing}
                  disabled={isProcessing}>
                  <Play className="mr-2" /> Start
                </button>
              </ArcherElement>
            </div>

            {/* Second Column: Models */}
            <div className="w-1/3 relative">
              <div className="flex flex-col space-y-4">
                {[0, 1, 2, 3].map(index => (
                  <ArcherElement key={index} id={`model-${index}`}>
                    <Model index={index} isProcessing={isProcessing} />
                  </ArcherElement>
                ))}
              </div>
              {isProcessing && (
                <motion.div
                  className="absolute left-0 w-4 h-4 bg-yellow-400 rounded-full"
                  animate={{
                    y: ["0%", "100%"],
                    x: ["-50%", "-50%", "-100%", "0%", "-50%"],
                  }}
                  transition={{
                    duration: 2,
                    ease: "easeInOut",
                    times: [0, 0.4, 0.5, 0.9, 1],
                    repeat: Infinity,
                  }}
                />
              )}
            </div>

            {/* Third Column: Results */}
            <div className="w-1/3 pl-4">
              <div className="bg-white rounded p-4">
                <h2 className="text-xl font-bold mb-4">Results</h2>
                {results.map((result, index) => (
                  <div key={index} className="mb-2">
                    <span className="font-semibold">Model {index + 1}:</span>
                    <span className="text-green-500 ml-2">
                      {result || "Pending..."}
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
