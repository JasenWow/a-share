declare module 'asciichart' {
  interface PlotOptions {
    height?: number;
    colors?: string[];
    format?: (x: number) => string;
  }

  function plot(series: number | number[] | number[][], options?: PlotOptions): string;

  export const blue: string;
  export const green: string;
  export const red: string;
  export const yellow: string;
  export const cyan: string;
  export const reset: string;

  export { plot };
  export default { plot, blue, green, red, yellow, cyan, reset };
}
