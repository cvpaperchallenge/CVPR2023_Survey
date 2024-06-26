import { ColumnDef } from "@tanstack/react-table"
import { LuArrowUpDown } from "react-icons/lu";
import { Button } from "@/components/ui/button"

import { PaperInfo } from "@/lib/types"
import { RxFile, RxGlobe } from "react-icons/rx";
import { ScrollArea } from "@/components/ui/scroll-area";

import { useRouter } from 'next/navigation'

export const columns: ColumnDef<PaperInfo>[] = [
  {
    accessorKey: "title",
    header: ({ column }) => {
      return (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Title
          <LuArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      )
    },
    cell: ({ row }) => {
      const router = useRouter()
      const authors: string[] = row.getValue("authors")
      return (
        <div className="flex flex-col justify-start">
          <a onClick={() => router.push(`/details?conference=${row.original.conference}&id=${row.index}`)} className=" rounded px-2 py-1 cursor-pointer hover:bg-accent hover:text-accent-foreground active:bg-primary active:text-primary-foreground transition-colors">
            <div className="font-semibold">{row.getValue("title")}</div>
          </a>
          <ScrollArea className="h-10 mt-1">
            <div className="text-sm text-muted-foreground px-2 pb-2 ">{authors.join(", ")}</div>
          </ScrollArea>
        </div>
      )
    }
  },
  {
    accessorKey: "authors",
    header: "Authors",
    cell: ({ row }) => {
      const authors = row.getValue("authors") as string[]
      return <div>{authors.join(", ")}</div>
    },
    filterFn: (row, id, filterValue) => {
      // console.log(`Filtering authors with value: ${filterValue}`)
      // console.log(rows)
      const authorsArray = row.original.authors
      // Return true if authorsArray contains any author that includes filterValue
      return authorsArray.some((author) => filterValue.includes(author))
      // return rows.filter((row) => {
      //   const authors = row.values[id] as string[]
      //   return authors.some((author) => author.includes(filterValue))
      // })
    }
  },
  {
    accessorKey: "Link",
    header: () => <div className="text-center">Link</div>,
    cell : ({ row }) => {
      return (
        <div className="flex flex-col items-center gap-1.5">
          <a href={row.original.cvfLink} target="_blank" rel="noreferrer">
            <Button
              variant="outline"
              className="
              gap-1
              text-[var(--jade-11)]
              bg-[var(--jade-4)]
              hover:bg-[var(--jade-5)]
              hover:text-[var(--jade-12)]
              border-[var(--jade-6)]
              px-2
              h-6
              text-xs
            ">
              <RxGlobe/> CVF
            </Button>
          </a>
          <a href={row.original.pdfLink} target="_blank" rel="noreferrer">
            <Button
              variant="outline"
              className="
                gap-1
                text-[var(--red-11)]
                bg-[var(--red-3)]
                hover:bg-[var(--red-4)]
                hover:text-[var(--red-12)]
                border-[var(--red-6)]
                px-2
                h-6
                text-xs
            ">
                <RxFile/> PDF
            </Button>
          </a>
        </div>
      )
    },
  },
  {
    accessorKey: "conference",
    header: "Conference",
    cell: ({ row }) => {
      return (
        <div className="text-center">
          {row.original.conference}
        </div>
      )
    },
    filterFn: (row, id, filterValue) => {
      return filterValue.includes(row.getValue(id))
    },
  },
]