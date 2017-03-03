#!/usr/bin/perl

while (my $line = <STDIN>)
{
  $line =~ s/[\r\n]//g;
  $line =~ s/\s+$//g;
  $line =~ s/^\s+//g;
  $line =~ s/\<\$number\>/\$number/g;
  $line =~ s/\<\$time\>/\$time/g;
  $line =~ s/\<\$date\>/\$date/g;
  print STDOUT $line."\n";
}

