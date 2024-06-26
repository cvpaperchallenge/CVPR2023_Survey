'use client'

import { columns } from './columns'
import { DataTable } from '@/components/data-table'
import { useState, useEffect } from 'react'
import { getPaperList, handleFetchArrayResult } from '@/lib/fetch'
import { PaperInfo } from '@/lib/types'

const loadPaperLists = async () => {
  const result = await getPaperList();
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');
};

export default function ListPage() {
  const [papers, setPapers] = useState<PaperInfo[]>([])

  useEffect(() => {
    const fetchPapers = async () => {
      setPapers(await loadPaperLists())
    }
    fetchPapers()
  }, [])

  return (
    <div className="container mx-auto py-10 flex flex-col items-center">
      <DataTable columns={columns} data={papers} />
    </div>
  )
}
