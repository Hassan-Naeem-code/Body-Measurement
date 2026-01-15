'use client';

import { useState } from 'react';
import { Download, FileText, Loader2 } from 'lucide-react';
import { exportMeasurementsToPDF, exportSinglePersonPDF } from '@/lib/pdf-export';
import type { MultiPersonMeasurementResult, PersonMeasurement } from '@/lib/types';
import { toast } from 'sonner';

interface PDFExportButtonProps {
  result?: MultiPersonMeasurementResult;
  singlePerson?: PersonMeasurement;
  brandName?: string;
  variant?: 'primary' | 'secondary' | 'icon';
  className?: string;
}

export function PDFExportButton({
  result,
  singlePerson,
  brandName,
  variant = 'primary',
  className = '',
}: PDFExportButtonProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    if (!result && !singlePerson) {
      toast.error('No measurement data to export');
      return;
    }

    setIsExporting(true);
    try {
      if (singlePerson) {
        await exportSinglePersonPDF(singlePerson, { brandName });
      } else if (result) {
        await exportMeasurementsToPDF(result, { brandName });
      }
      toast.success('PDF report downloaded successfully');
    } catch (error) {
      console.error('PDF export error:', error);
      toast.error('Failed to generate PDF report');
    } finally {
      setIsExporting(false);
    }
  };

  if (variant === 'icon') {
    return (
      <button
        onClick={handleExport}
        disabled={isExporting}
        className={`flex items-center justify-center w-10 h-10 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-50 ${className}`}
        title="Export to PDF"
      >
        {isExporting ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : (
          <Download className="w-5 h-5" />
        )}
      </button>
    );
  }

  if (variant === 'secondary') {
    return (
      <button
        onClick={handleExport}
        disabled={isExporting}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-card text-foreground hover:bg-muted transition-colors disabled:opacity-50 ${className}`}
      >
        {isExporting ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Generating...</span>
          </>
        ) : (
          <>
            <FileText className="w-4 h-4" />
            <span>Export PDF</span>
          </>
        )}
      </button>
    );
  }

  return (
    <button
      onClick={handleExport}
      disabled={isExporting}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 ${className}`}
    >
      {isExporting ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>Generating PDF...</span>
        </>
      ) : (
        <>
          <Download className="w-4 h-4" />
          <span>Download Report</span>
        </>
      )}
    </button>
  );
}
