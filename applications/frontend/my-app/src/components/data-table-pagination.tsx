import { Table } from '@tanstack/react-table'
import {
  RxCaretLeft,
  RxCaretRight,
  RxDoubleArrowLeft,
  RxDoubleArrowRight,
} from 'react-icons/rx'

import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
} from '@/components/ui/pagination'

interface DataTablePaginationProps<TData> {
  table: Table<TData>
  numPagesDisplayed: number
}

export function DataTablePagination<TData>({
  table,
  numPagesDisplayed,
}: DataTablePaginationProps<TData>) {
  const getPaginationItems = () => {
    if (numPagesDisplayed === 0) {
      return []
    }
    const pageCount = table.getPageCount()
    const pageIndex = table.getState().pagination.pageIndex + 1
    const items = []
    let startPage = Math.max(1, pageIndex - Math.floor(numPagesDisplayed / 2))
    let endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1)

    // Adjust start and end if we're too close to the boundaries
    if (startPage === 1) {
      endPage = Math.min(pageCount, startPage + numPagesDisplayed - 1)
    } else if (endPage === pageCount) {
      startPage = Math.max(1, endPage - numPagesDisplayed + 1)
    }
    // Add ellipsis if necessary at the beginning
    if (startPage > 1 || numPagesDisplayed < pageCount) {
      items.push(
        <PaginationItem key="start-ellipsis">
          <PaginationEllipsis />
        </PaginationItem>
      )
    }

    for (let i = startPage; i <= endPage; i++) {
      items.push(
        <PaginationItem>
          <PaginationLink
            isActive={i === pageIndex}
            onClick={() => {
              table.setPageIndex(i - 1)
            }}
          >
            {i}
          </PaginationLink>
        </PaginationItem>
      )
    }

    // Add ellipsis if necessary at the end
    if (endPage < pageCount || numPagesDisplayed < pageCount) {
      items.push(
        <PaginationItem key="end-ellipsis">
          <PaginationEllipsis />
        </PaginationItem>
      )
    }
    return items
  }

  return (
    <Pagination className="border-t bg-[var(--teal-4)] p-3 dark:bg-[var(--teal-3)]">
      <PaginationContent>
        <PaginationItem>
          <PaginationLink
            aria-label="Go to the first page"
            className="gap-1 p-3"
            isDisabled={!table.getCanPreviousPage()}
            onClick={() => table.firstPage()}
            size="default"
          >
            <RxDoubleArrowLeft className="size-4" />
          </PaginationLink>
        </PaginationItem>
        <PaginationItem>
          <PaginationLink
            aria-label="Go to previous page"
            className="gap-1 p-3"
            isDisabled={!table.getCanPreviousPage()}
            onClick={() => table.previousPage()}
            size="default"
          >
            <RxCaretLeft className="size-4" />
          </PaginationLink>
        </PaginationItem>
        {getPaginationItems()}
        <PaginationItem>
          <PaginationLink
            aria-label="Go to next page"
            className="gap-1 p-3"
            isDisabled={!table.getCanNextPage()}
            onClick={() => table.nextPage()}
            size="default"
          >
            <RxCaretRight className="size-4" />
          </PaginationLink>
        </PaginationItem>
        <PaginationItem>
          <PaginationLink
            aria-label="Go to the last page"
            className="gap-1 p-3"
            isDisabled={!table.getCanNextPage()}
            onClick={() => table.lastPage()}
            size="default"
          >
            <RxDoubleArrowRight className="size-4" />
          </PaginationLink>
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  )
}
