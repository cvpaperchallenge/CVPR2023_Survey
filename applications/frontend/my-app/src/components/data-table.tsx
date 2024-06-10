"use client"

import { useState } from "react"
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

import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

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
    console.log(`startPage: ${startPage}`)
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

  const router = useRouter()

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
      <Button variant="ghost">Authors</Button>
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
