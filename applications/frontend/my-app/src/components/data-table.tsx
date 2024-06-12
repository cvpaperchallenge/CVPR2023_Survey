"use client"

import { useState } from "react"
import { DropdownMenuCheckboxItemProps } from "@radix-ui/react-dropdown-menu"

import { useRouter } from 'next/navigation'
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"

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

type Checked = DropdownMenuCheckboxItemProps["checked"]

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
  // set default values
  numPagesDisplayed?: number
}

export function DataTable<TData, TValue>({
  columns,
  data,
  numPagesDisplayed = 5,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>(
    []
  )
  const router = useRouter()

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    onColumnFiltersChange: setColumnFilters,
    getFilteredRowModel: getFilteredRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
    state: {
      sorting,
      columnFilters,
    },
  })

  const getPaginationItems = () => {
    const pageCount = table.getPageCount();
    const pageIndex = table.getState().pagination.pageIndex + 1;
    const items = [];
    let startPage = Math.max(1, pageIndex - Math.floor(numPagesDisplayed / 2));
    let endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1);

    // Adjust start and end if we're too close to the boundaries
    if (startPage === 1) {
      endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1);
    } else if (endPage === pageCount) {
      startPage = Math.max(1, endPage - numPagesDisplayed + 1);
    }
    // Add ellipsis if necessary at the beginning
    if (startPage > 1) {
      items.push(
          <PaginationItem key="start-ellipsis">
              <PaginationEllipsis />
          </PaginationItem>
      );
    }

    for (let i = startPage; i <= endPage; i++) {
        items.push(
            <PaginationItem>
                <PaginationLink
                    isActive={i === pageIndex}
                    onClick={() => {
                        table.setPageIndex(i - 1);
                    }}
                >
                    {i}
                </PaginationLink>
            </PaginationItem>
        );
    }

    // Add ellipsis if necessary at the end
    if (endPage < pageCount) {
      items.push(
          <PaginationItem key="end-ellipsis">
              <PaginationEllipsis />
          </PaginationItem>
      );
    }
    return items;
  };

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
    <div>
      <div className="flex items-center py-4">
        <Input
          placeholder="Filter titles..."
          value={(table.getColumn("title")?.getFilterValue() as string) ?? ""}
          onChange={(event) =>
            table.getColumn("title")?.setFilterValue(event.target.value)
          }
          className="max-w-sm"
        />
      </div>
      {getMultiSelectItemsFilter("authors")}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    return (
                      <TableHead key={header.id}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                      </TableHead>
                    )
                  })}
                </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                    onClick={() => router.push(`/dashboard/paper/${row.id}`)}
                    className="hover:bg-accent hover:text-accent-foreground active:bg-primary active:text-primary-foreground transition-colors"
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center">
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                href="#"
                onClick={() => table.previousPage()}
              />
            </PaginationItem>
            {getPaginationItems()}
            <PaginationItem>
              <PaginationNext
                href="#"
                onClick={
                  table.getCanNextPage() ? (
                    () => table.nextPage()
                  ) : (
                    () => {}
                  )}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </div>
    </div>
  )
}
