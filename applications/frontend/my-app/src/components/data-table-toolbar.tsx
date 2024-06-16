
import { Input } from "@/components/ui/input"

import { Table } from "@tanstack/react-table"
import { DataTableFacetedFilter } from "./data-table-faceted-filter"

interface DataTableToolbarProps<TData> {
  table: Table<TData>
}

export function DataTableToolbar<TData>({
  table,
}: DataTableToolbarProps<TData>) {
  return (
    <div className="flex flex-col justify-start gap-2 py-4">
      <Input
        placeholder="Filter titles..."
        type="text"
        value={(table.getColumn("title")?.getFilterValue() as string) ?? ""}
        onChange={(event) =>
          table.getColumn("title")?.setFilterValue(event.target.value)
        }
        className="max-w-sm"
      />
      <div className="flex flex-row items-center gap-2">
        <DataTableFacetedFilter table={table} columnName="authors"/>
        <DataTableFacetedFilter table={table} columnName="conference"/>
      </div>
    </div>
  )
}