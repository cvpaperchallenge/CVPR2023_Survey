'use client'

import * as React from 'react'

import { columns } from './columns'
import { DataTable } from '@/components/data-table'
import { useRef, useState, type RefObject, useEffect } from 'react'
import { getPaperList, searchPapers, handleFetchArrayResult } from '@/lib/fetch'
import { PaperInfo } from '@/lib/types'

const generateDummyData = (num: number): PaperInfo[] => {
  const papers: PaperInfo[] = [];

  for (let i = 1; i <= num; i++) {
    papers.push({
      title: `Loooooooooooooooooooong Title ${i}`,
      authors: [
        `Author ${String(i).padStart(3, '0')}`,
        `Author ${String(i + 1).padStart(3, '0')}`,
        `Author ${String(i + 2).padStart(3, '0')}`
      ],
      cvfLink: `CVFLink ${i}`,
      pdfLink: `PDFLink ${i}`,
      conference: `Conference ${i}`
    });
  }

  return papers;
}

const loadPaperLists = async () => {
  const result = await getPaperList();
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');
};

export default function ModeToggle() {
  const [papers, setPapers] = useState<PaperInfo[]>([])

  useEffect(() => {
    const fetchPapers = async () => {
      setPapers(await loadPaperLists())
    }
    fetchPapers()
  }, [])

  return (
    <div>
      <div className="container mx-auto py-10">
        <DataTable columns={columns} data={papers} />
      </div>
    </div>
  )
}
