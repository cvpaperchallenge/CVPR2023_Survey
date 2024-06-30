import { ColumnDef } from '@tanstack/react-table'
import { useRouter } from 'next/navigation'
import { LuArrowUpDown } from 'react-icons/lu'
import { RxFile, RxGlobe } from 'react-icons/rx'

import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'

import { PaperInfo } from '@/lib/types'

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
    cell: ({ row }) => {
      const router = useRouter()
      const authors: string[] = row.getValue('authors')
      return (
        <div className="flex flex-col justify-start">
          <a
            className=" cursor-pointer rounded px-2 py-1 transition-colors hover:bg-accent hover:text-accent-foreground active:bg-primary active:text-primary-foreground"
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
    },
  },
  {
    accessorKey: 'authors',
    header: 'Authors',
    cell: ({ row }) => {
      const authors: string[] = row.getValue('authors')
      return <div>{authors.join(', ')}</div>
    },
    filterFn: (row, id, filterValue) => {
      // console.log(`Filtering authors with value: ${filterValue}`)
      // console.log(rows)
      const authorsArray = row.original.authors
      // Return true if authorsArray contains any author that includes filterValue
      return authorsArray.some((author) => (filterValue as string[]).includes(author))
      // return rows.filter((row) => {
      //   const authors = row.values[id] as string[]
      //   return authors.some((author) => author.includes(filterValue))
      // })
    },
  },
  {
    accessorKey: 'link',
    header: () => <div className="text-center">Link</div>,
    cell: ({ row }) => {
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
    },
  },
  {
    accessorKey: 'conference',
    header: 'Conference',
    cell: ({ row }) => {
      return <div className="text-center">{row.original.conference}</div>
    },
    filterFn: (row, id, filterValue) => {
      return filterValue.includes(row.getValue(id))
    },
  },
]
