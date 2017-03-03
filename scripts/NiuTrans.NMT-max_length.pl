#!/usr/bin/perl -w 

$line_num = 0;
$max_len = 0;
$max_len_num = 0;
while(my $line = <STDIN>)
{
  ++$line_num;
  $line =~ s/[\r\n]//g;
  my @words = split /\s/, $line;
  if (scalar(@words) > $max_len) 
  {
    $max_len = scalar(@words);
    $max_len_num = $line_num;
  }
}

print STDERR "MAX_LEN=$max_len MAX_LEN_NUM=$max_len_num\n";

