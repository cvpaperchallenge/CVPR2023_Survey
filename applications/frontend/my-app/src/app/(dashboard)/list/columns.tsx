import { ColumnDef, Row } from '@tanstack/react-table'
import { useRouter } from 'next/navigation'
import { LuArrowUpDown } from 'react-icons/lu'
import { RxFile, RxGlobe } from 'react-icons/rx'

import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'

import { PaperInfo } from '@/lib/types'

const TitleCell = ({ row }: { row: Row<PaperInfo> }) => {
  const router = useRouter()
  const authors: string[] = row.getValue('authors')
  return (
    <div className="flex flex-col justify-start">
      <a
        className="cursor-pointer rounded px-2 py-1 transition-colors hover:bg-accent hover:text-accent-foreground active:bg-primary active:text-primary-foreground"
        onClick={() => router.push(`/details?id=${row.index}`)}
      >
        <div className="font-semibold">{row.getValue('title')}</div>
      </a>
      <ScrollArea className="mt-1 h-10">
        <div className="px-2 pb-2 text-sm text-muted-foreground ">
          {authors.join(', ')}
        </div>
      </ScrollArea>
    </div>
  )
}

const AuthorsCell = ({ row }: { row: Row<PaperInfo> }) => {
  const authors: string[] = row.getValue('authors')
  return <div>{authors.join(', ')}</div>
}

const LinkCell = ({ row }: { row: Row<PaperInfo> }) => {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <a href={row.original.cvfLink} rel="noreferrer" target="_blank">
        <Button
          className="
            h-6
            gap-1
            border-[var(--jade-6)]
            bg-[var(--jade-4)]
            px-2
            text-xs
            text-[var(--jade-11)]
            hover:bg-[var(--jade-5)]
            hover:text-[var(--jade-12)]
          "
          variant="outline"
        >
          <RxGlobe /> CVF
        </Button>
      </a>
      <a href={row.original.pdfLink} rel="noreferrer" target="_blank">
        <Button
          className="
            h-6
            gap-1
            border-[var(--red-6)]
            bg-[var(--red-3)]
            px-2
            text-xs
            text-[var(--red-11)]
            hover:bg-[var(--red-4)]
            hover:text-[var(--red-12)]
          "
          variant="outline"
        >
          <RxFile /> PDF
        </Button>
      </a>
    </div>
  )
}

const ConferenceCell = ({ row }: { row: Row<PaperInfo> }) => {
  return <div className="text-center">{row.original.conference}</div>
}

export const columns: ColumnDef<PaperInfo>[] = [
  {
    accessorKey: 'title',
    header: ({ column }) => {
      return (
        <Button
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
          variant="ghost"
        >
          Title
          <LuArrowUpDown className="ml-2 size-4" />
        </Button>
      )
    },
    cell: TitleCell,
  },
  {
    accessorKey: 'authors',
    header: 'Authors',
    cell: AuthorsCell,
    filterFn: (row, id, filterValue) => {
      const authorsArray = row.original.authors
      return authorsArray.some((author) =>
        (filterValue as string[]).includes(author)
      )
    },
  },
  {
    accessorKey: 'link',
    header: () => <div className="text-center">Link</div>,
    cell: LinkCell,
  },
  {
    accessorKey: 'conference',
    header: 'Conference',
    cell: ConferenceCell,
    filterFn: (row, id, filterValue) => {
      return (filterValue as string[]).includes(row.getValue(id))
    },
  },
]
