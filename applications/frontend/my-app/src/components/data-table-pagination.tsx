import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"

import { Table } from "@tanstack/react-table"

interface DataTablePaginationProps<TData> {
  table: Table<TData>
  numPagesDisplayed: number
}

export function DataTablePagination<TData>({
  table,
  numPagesDisplayed
}: DataTablePaginationProps<TData>) {
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
  }

  return (
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
  )
}