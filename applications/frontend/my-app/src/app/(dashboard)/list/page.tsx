'use client'

import { columns } from './columns'
import { DataTable } from '@/components/data-table'
import { useState, useEffect } from 'react'
import { getPaperList, handleFetchArrayResult } from '@/lib/fetch'
import { PaperInfo } from '@/lib/types'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"

const loadPaperLists = async () => {
  const result = await getPaperList();
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');
};

export default function ListPage() {
  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth);
    };
    const fetchPapers = async () => {
      setPapers(await loadPaperLists())
    }
    fetchPapers()

    window.addEventListener('resize', handleResize);

    // 初回レンダリング時に画面の幅を設定
    handleResize();

    // コンポーネントがアンマウントされる時にクリーンアップ
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [])

  // 画面の幅に応じて変数の値を設定
  const numPagesDisplayed = width > 600 ? 5 : width > 490 ? 3 : 0;

  return (
    <div className="flex flex-col items-center gap-12 w-screen">
      <div className='flex flex-row justify-start w-full px-10'>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>Paper List</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <DataTable columns={columns} data={papers} width={width} numPagesDisplayed={numPagesDisplayed}/>
    </div>
  )
}
