'use client'

import * as React from 'react'

import { Paper, columns } from './columns'
import { DataTable } from '@/components/data-table'
import { useRef, useState, type RefObject, useEffect } from 'react'

const generateDummyData = (num: number): Paper[] => {
  const papers: Paper[] = [];

  for (let i = 1; i <= num; i++) {
    papers.push({
      id: i.toString(),
      title: `Loooooooooooooooooooong Title ${i}`,
      authors: [
        `Author ${String(i).padStart(3, '0')}`,
        `Author ${String(i + 1).padStart(3, '0')}`,
        `Author ${String(i + 2).padStart(3, '0')}`
      ],
      cvfLink: `Link ${i}`,
      conference: `Conference ${i}`
    });
  }

  return papers;
}

const loadPaperLists = async () => {
  // const result = await getPaperList(conference);
  // return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');

  return generateDummyData(100);
};

export default function ModeToggle() {
  const [papers, setPapers] = useState<Paper[]>([])

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
