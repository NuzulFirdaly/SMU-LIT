import * as React from 'react';
import RouterLink from 'next/link';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import { config } from '@/config';
import { paths } from '@/paths';

export const metadata = { title: `Not found | ${config.site.name}` };

export default function NotFound() {
  return (
    <Box
      component="main"
      sx={{
        alignItems: 'center',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        minHeight: '100%',
        py: '64px',
      }}
    >
      <Container maxWidth="lg">
        <Stack spacing={6}>
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <Box
              alt="Not found"
              component="img"
              src="/assets/not-found.svg"
              sx={{ height: 'auto', maxWidth: '100%', width: '200px' }}
            />
          </Box>
          <Stack spacing={1} sx={{ textAlign: 'center' }}>
            <Typography variant="h4">The content you are accessing may not be moderated</Typography>
            <Typography color="text.secondary">
              The content you are about to view (sammyboy.com) has not been moderated and may contain material that
              could be harmful or distressing. Please proceed with caution. If you feel uncomfortable, consider
              navigating away or seeking support. <Link href="https://www.sos.org.sg/">Learn more</Link>
              <br />
              <br />
              This page has been enforced by your Internet Service Provider.
            </Typography>
          </Stack>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <div></div>
            <Button component={RouterLink} href={paths.home} variant="contained">
              Back to home
            </Button>
            <Button component={RouterLink} href={paths.home} variant="contained">
              Continue to website
            </Button>
            <div></div>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}
