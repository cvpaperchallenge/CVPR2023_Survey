import { ColumnDef } from "@tanstack/react-table"
import { LuArrowUpDown } from "react-icons/lu";

import { Button } from "@/components/ui/button"

import { PaperInfo } from "@/lib/types"
import { Badge } from "@/components/ui/badge";
import { RxGlobe } from "react-icons/rx";

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
  },
  {
    accessorKey: "authors",
    header: "Authors",
    cell: ({ row }) => {
      const authors = row.getValue("authors") as string[]
      return <div>{authors.join(", ")}</div>
    },
    filterFn: (rows, id, filterValue) => {
      // console.log(`Filtering authors with value: ${filterValue}`)
      // console.log(rows)
      const authorsArray = rows.original.authors
      // Return true if authorsArray contains any author that includes filterValue
      return authorsArray.some((author) => filterValue.includes(author))
      // return rows.filter((row) => {
      //   const authors = row.values[id] as string[]
      //   return authors.some((author) => author.includes(filterValue))
      // })
    }
  },
  {
    accessorKey: "cvfLink",
    header: "Link",
    cell : ({ row }) => {
      return (
        // <div className="flex space-x-2">
        //   <Badge variant="outline">{row.original.label}</Badge>
        //   <span className="max-w-[500px] truncate font-medium">
        //     {row.getValue("title")}
        //   </span>
        // </div>
        <a href={row.original.cvfLink} target="_blank" rel="noreferrer">
          <Badge variant="outline" className="gap-1"><RxGlobe/> CVF</Badge>
        </a>
      )
    },
  },
  {
    accessorKey: "conference",
    header: "Conference",
  },
]