import React from 'react';
import { Box, Text } from 'ink';

interface AppProps {
  command?: string;
}

export function App({ command }: AppProps) {
  return (
    <Box flexDirection="column" padding={1}>
      <Text color="cyan" bold>CLI Viz</Text>
      {command && <Text dimColor>Running: {command}</Text>}
    </Box>
  );
}