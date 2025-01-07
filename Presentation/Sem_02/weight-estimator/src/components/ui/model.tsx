import {Loader2} from "lucide-react";

interface ModelProps {
  name: string;
  isProcessing: boolean;
}

export default function Model({name, isProcessing}: ModelProps) {
  return (
    <div
      className={`bg-white p-4 rounded relative flex justify-center items-center text-black`}>
      <div>{name}</div>
      {isProcessing && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded">
          <Loader2 className="animate-spin text-white" />
        </div>
      )}
    </div>
  );
}
