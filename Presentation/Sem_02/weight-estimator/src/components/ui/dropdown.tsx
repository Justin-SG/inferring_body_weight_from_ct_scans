"use client";

import {useState} from "react";
import {ChevronDown} from "lucide-react";

interface DropdownProps {
  options: string[];
  selected: string;
  onSelect: (option: string) => void;
}

export default function Dropdown({options, selected, onSelect}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        className="w-full bg-white text-black p-2 rounded flex justify-between items-center"
        onClick={() => setIsOpen(!isOpen)}>
        {selected} <ChevronDown />
      </button>
      {isOpen && (
        <div className="absolute top-full left-0 w-full bg-white rounded mt-1">
          {options.map(option => (
            <div
              key={option}
              className="p-2 hover:bg-gray-200 cursor-pointer"
              onClick={() => {
                onSelect(option);
                setIsOpen(false);
              }}>
              {option}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
