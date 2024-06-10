import { ColumnDef } from "@tanstack/react-table"
import { LuArrowUpDown } from "react-icons/lu";

import { Button } from "@/components/ui/button"

export interface Paper {
  id: string
  title: string
  authors: string[]
  cvfLink: string
  conference: string
}

export const columns: ColumnDef<Paper>[] = [
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
    }
  },
  {
    accessorKey: "cvfLink",
    header: "Link",
  },
  {
    accessorKey: "conference",
    header: "Conference",
  },
]