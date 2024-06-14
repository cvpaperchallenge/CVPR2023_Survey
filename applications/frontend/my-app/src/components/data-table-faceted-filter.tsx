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
import { ScrollArea } from "@/components/ui/scroll-area"

import { Table } from '@tanstack/react-table'

import { RxMagnifyingGlass } from "react-icons/rx";
import { useState } from "react"

interface DataTableFacetedFilterProps<TData> {
  table: Table<TData>
  columnName: string
}

export function DataTableFacetedFilter<TData>({
  table,
  columnName
}: DataTableFacetedFilterProps<TData>) {
  const [filterText, setFilterText] = useState<string>("")
  const column = table.getColumn(columnName)
  if (!column) return null

  const facets = new Set(
    Array.from(column?.getFacetedUniqueValues().keys()).flat()
  )
  const selectedValues = new Set<string>(column?.getFilterValue() as string[])

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost">{column.columnDef.header}</Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent loop={false}>
        <DropdownMenuLabel>Filter by {column.columnDef.header}</DropdownMenuLabel>
        <RxMagnifyingGlass className="h-4 w-4" />
        <Input
          placeholder="Search..."
          type="text"
          value={filterText}
          onChange={(event) => {
              event.stopPropagation()
              event.preventDefault()
              event.nativeEvent.stopImmediatePropagation()
              setFilterText(event.target.value)
            }
          }
          onKeyDown={(event) => {
            event.nativeEvent.stopImmediatePropagation()
          }}
        />
        <DropdownMenuSeparator />
        <ScrollArea className="h-72">
          {Array.from(facets).map((option) => {
            const isSelected = selectedValues.has(option) //.value)
            return (
              <DropdownMenuCheckboxItem
                key={option}
                checked={isSelected}
                onSelect={(e) => {
                  e.preventDefault()
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
              >
                {option}
              </DropdownMenuCheckboxItem>
            )
          })}
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}