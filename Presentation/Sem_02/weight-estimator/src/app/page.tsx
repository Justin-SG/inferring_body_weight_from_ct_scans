'use client'

import { useState } from 'react'
import { ChevronDown, Play, Loader2 } from 'lucide-react'
import { motion } from 'framer-motion'

const ctScans = ['Brain CT', 'Chest CT', 'Abdominal CT', 'Spine CT']

export default function Home() {
  const [selectedScan, setSelectedScan] = useState(ctScans[0])
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState(['', '', '', ''])

  const startProcessing = () => {
    setIsProcessing(true)
    setTimeout(() => {
      setResults(['Result 1', 'Result 2', 'Result 3', 'Result 4'])
      setIsProcessing(false)
    }, 3000)
  }

  return (
    <div className="min-h-screen bg-cover bg-center flex items-center justify-center" style={{backgroundImage: "url('https://images.unsplash.com/photo-1557683311-eac922347aa1?auto=format&fit=crop&q=80&w=2029&ixlib=rb-4.0.3')"}}> 
      <div className="bg-black bg-opacity-50 p-8 rounded-lg w-full max-w-2xl">
        <div className="flex">
          <div className="w-1/3 pr-4">
            <div className="relative mb-4">
              <button
                className="w-full bg-white text-black p-2 rounded flex justify-between items-center"
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              >
                {selectedScan} <ChevronDown />
              </button>
              {isDropdownOpen && (
                <div className="absolute top-full left-0 w-full bg-white rounded mt-1">
                  {ctScans.map((scan) => (
                    <div
                      key={scan}
                      className="p-2 hover:bg-gray-200 cursor-pointer"
                      onClick={() => {
                        setSelectedScan(scan)
                        setIsDropdownOpen(false)
                      }}
                    >
                      {scan}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button
              className="w-full bg-blue-500 text-white p-2 rounded flex items-center justify-center"
              onClick={startProcessing}
              disabled={isProcessing}
            >
              <Play className="mr-2" /> Start
            </button>
          </div>
          <div className="w-2/3 relative">
            <div className="flex flex-col space-y-4">
              {[0, 1, 2, 3].map((index) => (
                <div key={index} className="bg-white p-4 rounded relative flex justify-between items-center">
                  <div>Model {index + 1}</div>
                  {isProcessing && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded">
                      <Loader2 className="animate-spin text-white" />
                    </div>
                  )}
                  {results[index] && (
                    <div className="text-green-500">{results[index]}</div>
                  )}
                </div>
              ))}
            </div>
            <svg className="absolute left-0 h-full w-8 -ml-4" style={{zIndex: -1}}>
              <line x1="50%" y1="0" x2="50%" y2="100%" stroke="white" strokeWidth="2" />
            </svg>
            {isProcessing && (
              <motion.div
                className="absolute left-0 w-4 h-4 bg-yellow-400 rounded-full"
                animate={{
                  y: ['0%', '100%'],
                  x: ['-50%', '-50%', '-100%', '0%', '-50%'],
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
        </div>
      </div>
    </div>
  )
}
