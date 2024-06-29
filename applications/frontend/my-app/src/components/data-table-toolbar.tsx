import { Table } from '@tanstack/react-table'

import { Input } from '@/components/ui/input'

import { DataTableFacetedFilter } from './data-table-faceted-filter'

interface DataTableToolbarProps<TData> {
  table: Table<TData>
}

export function DataTableToolbar<TData>({
  table,
}: DataTableToolbarProps<TData>) {
  return (
    <div className="flex flex-col justify-start gap-2 py-4">
      <Input
        className="max-w-sm"
        onChange={(event) =>
          table.getColumn('title')?.setFilterValue(event.target.value)
        }
        placeholder="Filter titles..."
        type="text"
        value={(table.getColumn('title')?.getFilterValue() as string) ?? ''}
      />
      <div className="flex flex-row items-center gap-2">
        <DataTableFacetedFilter columnName="authors" table={table} />
        <DataTableFacetedFilter columnName="conference" table={table} />
      </div>
    </div>
  )
}
