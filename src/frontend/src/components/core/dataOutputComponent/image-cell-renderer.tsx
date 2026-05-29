import { getBaseUrl } from "@/customization/utils/urls";

export const ImageCellRenderer = (props: any) => {
  const value = props.value;

  // Check if value looks like a file path (contains / and potentially flow_id pattern)
  if (!value || typeof value !== "string") {
    return <span>{value}</span>;
  }

  // Match patterns like "flow_id/ImageAnnotator/filename.jpg" or "flow_id/filename.jpg"
  const filePathPattern = /^[a-f0-9\-]+\/[\w\/\-\.]+\.(jpg|jpeg|png|gif|webp)$/i;

  if (filePathPattern.test(value)) {
    const imageSrc = `${getBaseUrl()}files/images/${value}`;
    return (
      <div className="flex h-full items-center justify-center p-2">
        <img
          src={imageSrc}
          alt="Output"
          style={{
            maxHeight: "300px",
            maxWidth: "300px",
            objectFit: "contain",
            borderRadius: "4px",
          }}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
          }}
        />
      </div>
    );
  }

  return <span title={value}>{value}</span>;
};
