import {Loader2} from "lucide-react";

interface ModelProps {
  index: number;
  isProcessing: boolean;
}

export default function Model({index, isProcessing}: ModelProps) {
  return (
    <div
      className={`model-${index} bg-white p-4 rounded relative flex justify-between items-center`}>
      <div>Model {index + 1}</div>
      {isProcessing && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded">
          <Loader2 className="animate-spin text-white" />
        </div>
      )}
    </div>
  );
}
