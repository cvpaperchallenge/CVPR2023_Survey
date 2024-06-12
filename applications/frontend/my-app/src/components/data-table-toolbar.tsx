import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

import { Table } from "@tanstack/react-table"

interface DataTableToolbarProps<TData> {
  table: Table<TData>
}

export function DataTableToolbar<TData>({
  table
}: DataTableToolbarProps<TData>) {
  const column = table.getColumn("authors")
  // const autoCompleteSuggestions =  Array.from(
  //   new Set(
  //     Array.from(column.getFacetedUniqueValues().keys()).flat()
  //   )
  // ).sort().slice(0, 5000);
  const facets = new Set(
    Array.from(column?.getFacetedUniqueValues().keys()).flat()
  )
  const selectedValues = new Set<string>(column?.getFilterValue() as string[])

  console.log(`column header: ${column.columnDef.header}`)
  console.log(`column filtering values: ${column.getFilterValue()}`)

  const getMultiSelectItemsFilter = (columnName: string) => {
    const column = table.getColumn(columnName)
    if (!column) return null
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost">{column.columnDef.header}</Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuLabel>Filter by {column.columnDef.header}</DropdownMenuLabel>
          <DropdownMenuSeparator />
          {Array.from(facets).map((option) => {
            const isSelected = selectedValues.has(option) //.value)
            return (
              <DropdownMenuCheckboxItem
                key={option}
                checked={isSelected}
                onSelect={() => {
                  // すでに選択されている項目をクリックした場合はselectedValuesから削除
                  if (isSelected) {
                    selectedValues.delete(option)
                  // 選択されていない項目をクリックした場合は場合はselectedValuesに追加
                  } else {
                    selectedValues.add(option)
                  }
                  // selectedValuesに登録された値をフィルター値に設定
                  column?.setFilterValue(
                    Array.from(selectedValues).length ? Array.from(selectedValues) : undefined
                  )
                }}
                // onCheckedChange={(checked) => {
                //   column.setFilterValue(
                //     // checked
                //     //   ? [...((column.getFilterValue() as string[]) ?? []), author]
                //     //   : ((column.getFilterValue() as string[]) ?? []).filter((a) => a !== author)

                //   )
                // }}
              >
                {option}
              </DropdownMenuCheckboxItem>
            )
          })}
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  return (
    <div className="flex items-center py-4">
      <Input
        placeholder="Filter titles..."
        value={(table.getColumn("title")?.getFilterValue() as string) ?? ""}
        onChange={(event) =>
          table.getColumn("title")?.setFilterValue(event.target.value)
        }
        className="max-w-sm"
      />
      {getMultiSelectItemsFilter("authors")}
    </div>
  )
}