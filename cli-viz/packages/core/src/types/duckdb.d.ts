declare module '@evan/duckdb' {
  export function open(path: string): {
    connect(): { query(sql: string): unknown[]; close(): void };
    close(): void;
  };
}