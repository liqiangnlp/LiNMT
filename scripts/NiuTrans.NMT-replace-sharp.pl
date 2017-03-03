#!/usr/bin/perl
while (my $line = <STDIN>)
{
  $line =~ s/[\r\n]//g;
  $line =~ s/\#/sharp/g;
  print $line."\n";
}




